#!/usr/bin/env python
"""
Comprehensive Active Learning Experiment Runner.

This script:
1. Computes baseline models (trained on entire training set)
2. Runs AL simulations with all classifier + strategy combinations
3. Saves all results for visualization and analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add src to path so we can import the AL simulation
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.active_learning.simulate_embeddings import load_npz


def load_embeddings(embeddings_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train and test embeddings."""
    train_npz = embeddings_dir / "train_embeddings.npz"
    test_npz = embeddings_dir / "test_embeddings.npz"
    
    if not train_npz.exists() or not test_npz.exists():
        raise FileNotFoundError(f"Missing embeddings in {embeddings_dir}")
    
    X_train, y_train, _ = load_npz(train_npz)
    X_test, y_test, _ = load_npz(test_npz)
    return X_train, y_train, X_test, y_test


def compute_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: Path,
) -> dict[str, Any]:
    """Train baseline models on full training set and evaluate on test set."""
    print("\n" + "=" * 80)
    print("COMPUTING BASELINES (full training set)")
    print("=" * 80)
    
    classifiers = {
        "logreg": LogisticRegression(max_iter=2000, solver="lbfgs"),
        "random_forest": RandomForestClassifier(n_estimators=200),
        "svc": SVC(probability=True),
        "gbdt": GradientBoostingClassifier(n_estimators=200),
    }
    
    baseline_results = {}
    baseline_rows = []
    
    for name, clf in classifiers.items():
        print(f"\n[{name.upper()}] training on full set ({len(y_train)} samples)...")
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="macro"))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        
        baseline_results[name] = {
            "accuracy": acc,
            "f1_macro": f1,
            "precision_macro": prec,
            "recall_macro": rec,
            "n_train_samples": len(y_train),
        }
        
        baseline_rows.append({
            "classifier": name,
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4),
            "precision_macro": round(prec, 4),
            "recall_macro": round(rec, 4),
            "n_train_samples": len(y_train),
        })
        
        print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    
    # Save baseline results
    baseline_csv = out_dir / "baseline_results.csv"
    with baseline_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=baseline_rows[0].keys())
        writer.writeheader()
        writer.writerows(baseline_rows)
    
    print(f"\n[SAVE] Baseline results: {baseline_csv}")
    return baseline_results


def run_al_grid(
    embeddings_dir: Path,
    seed_size: int = 1,
    rounds: int = 50,
    query_size: int = 1,
    out_dir: Path | None = None,
    seed: int = 42,
) -> None:
    """Run AL experiments across all classifier + strategy combinations."""
    print("\n" + "=" * 80)
    print("RUNNING AL EXPERIMENTS")
    print("=" * 80)
    
    embeddings_dir = Path(embeddings_dir)
    out_dir = Path(out_dir or (embeddings_dir.parent / "al_experiments"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    X_train, y_train, X_test, y_test = load_embeddings(embeddings_dir)
    print(f"\nLoaded embeddings: X_train {X_train.shape}, X_test {X_test.shape}")
    
    # Compute baselines
    baselines = compute_baselines(X_train, y_train, X_test, y_test, out_dir)
    
    # Run AL with all combinations
    strategies = ["random", "entropy", "margin", "least_confidence"]
    diversity_opts = [False, True]
    
    classifiers = {
        "logreg": LogisticRegression(max_iter=2000, solver="lbfgs"),
        "random_forest": RandomForestClassifier(n_estimators=200),
        "svc": SVC(probability=True),
        "gbdt": GradientBoostingClassifier(n_estimators=200),
    }
    
    # Collect all AL results for comparison
    all_al_curves = []
    
    for strategy in strategies:
        for diversity in diversity_opts:
            div_tag = "div" if diversity else "nodiv"
            print(f"\n[STRATEGY] {strategy} + {div_tag}")
            
            # Run AL for this configuration
            al_subdir = out_dir / f"{strategy}_{div_tag}"
            al_subdir.mkdir(parents=True, exist_ok=True)
            
            np.random.seed(seed)
            n_samples = X_train.shape[0]
            idxs = np.arange(n_samples)
            np.random.shuffle(idxs)
            
            seed_size_adj = min(seed_size, n_samples)
            labeled_idxs = list(idxs[:seed_size_adj].tolist())
            unlabeled_idxs = list(idxs[seed_size_adj:].tolist())
            
            for clf_name, clf in classifiers.items():
                print(f"  [{clf_name}] ", end="", flush=True)
                
                labeled = labeled_idxs.copy()
                unlabeled = unlabeled_idxs.copy()
                metrics: list[dict[str, Any]] = []
                
                for r in range(1, rounds + 1):
                    if not labeled or not unlabeled:
                        break
                    
                    X_l = X_train[labeled]
                    y_l = y_train[labeled]
                    
                    unique_classes = len(np.unique(y_l))
                    can_train = unique_classes >= 2
                    
                    acc, f1 = 0.0, 0.0
                    if can_train:
                        clf.fit(X_l, y_l)
                        y_pred = clf.predict(X_test)
                        acc = float(accuracy_score(y_test, y_pred))
                        f1 = float(f1_score(y_test, y_pred, average="macro"))
                    
                    # Query
                    if unlabeled:
                        X_pool = X_train[unlabeled]
                        
                        if strategy == "random":
                            # Random selection
                            util = np.random.rand(len(unlabeled)).astype(np.float32)
                        elif can_train:
                            probs = clf.predict_proba(X_pool)
                            if strategy == "entropy":
                                util = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
                            elif strategy == "margin":
                                part = np.partition(-probs, 1, axis=1)
                                top1, top2 = -part[:, 0], -part[:, 1]
                                util = 1.0 - (top1 - top2)
                            elif strategy == "least_confidence":
                                util = 1.0 - probs.max(axis=1)
                            else:
                                util = np.ones(len(unlabeled), dtype=np.float32)
                        else:
                            # No model trained and not random: uniform utility
                            util = np.ones(len(unlabeled), dtype=np.float32)
                        
                        actual_k = min(query_size, len(unlabeled))
                        if diversity and can_train and strategy != "random":
                            # Greedy diversity (not for random or untrained models)
                            emb_norm = X_pool / np.clip(np.linalg.norm(X_pool, axis=1, keepdims=True), 1e-8, None)
                            util_norm = (util - util.min()) / max(1e-8, util.max() - util.min())
                            
                            chosen = [int(np.argmax(util))]
                            remaining = set(range(len(unlabeled)))
                            remaining.remove(chosen[0])
                            
                            while len(chosen) < min(actual_k, len(unlabeled)) and remaining:
                                rem_idx = np.array(sorted(remaining), dtype=np.int64)
                                sims = emb_norm[rem_idx] @ emb_norm[np.array(chosen)].T
                                max_sim = sims.max(axis=1)
                                div_score = 1.0 - max_sim
                                score = 0.7 * util_norm[rem_idx] + 0.3 * div_score
                                pick_local = int(np.argmax(score))
                                pick = int(rem_idx[pick_local])
                                chosen.append(pick)
                                remaining.remove(pick)
                            chosen_idxs = [unlabeled[i] for i in chosen]
                        else:
                            order = np.argsort(-util)[:actual_k]
                            chosen_idxs = [unlabeled[i] for i in order]
                        
                        labeled.extend(chosen_idxs)
                        unlabeled = [i for i in unlabeled if i not in chosen_idxs]
                    
                    metrics.append({
                        "round": r,
                        "labeled_size": len(labeled),
                        "pool_size": len(unlabeled),
                        "test_accuracy": acc,
                        "test_f1_macro": f1,
                    })
                
                # Save this classifier's AL curve
                metrics_csv = al_subdir / f"{clf_name}_metrics.csv"
                with metrics_csv.open("w", newline="", encoding="utf-8") as f:
                    fieldnames = ["round", "labeled_size", "pool_size", "test_accuracy", "test_f1_macro"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(metrics)
                
                # Add to global curves for comparison
                for metric in metrics:
                    all_al_curves.append({
                        "strategy": strategy,
                        "diversity": diversity,
                        "classifier": clf_name,
                        "round": metric["round"],
                        "labeled_size": metric["labeled_size"],
                        "test_accuracy": metric["test_accuracy"],
                        "test_f1_macro": metric["test_f1_macro"],
                    })
                
                print(f"ok ", end="", flush=True)
                
            print()  # newline after all classifiers for this strategy
    
    # Save aggregated curves for easy comparison
    curves_csv = out_dir / "all_al_curves.csv"
    with curves_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["strategy", "diversity", "classifier", "round", "labeled_size", "test_accuracy", "test_f1_macro"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_al_curves)
    
    print(f"\n[SAVE] All AL curves: {curves_csv}")
    
    # Save experiment metadata
    metadata = {
        "embeddings_dir": str(embeddings_dir),
        "seed_size": seed_size,
        "rounds": rounds,
        "query_size": query_size,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(out_dir),
    }
    with (out_dir / "experiment_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[SAVE] Experiment metadata: {out_dir / 'experiment_metadata.json'}")
    print("\n" + "=" * 80)
    print("AL EXPERIMENTS COMPLETED")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive AL experiment runner with baselines.")
    parser.add_argument("--embeddings-dir", type=Path, default=Path("artifacts/embedding_cnn/embeddings"))
    parser.add_argument("--seed-size", type=int, default=1, help="Initial labeled set size.")
    parser.add_argument("--rounds", type=int, default=50, help="Number of AL rounds.")
    parser.add_argument("--query-size", type=int, default=1, help="Number of samples per query.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for results.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    
    run_al_grid(
        embeddings_dir=args.embeddings_dir,
        seed_size=args.seed_size,
        rounds=args.rounds,
        query_size=args.query_size,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    
    elapsed = time.time() - started
    print(f"\nTotal experiment time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
