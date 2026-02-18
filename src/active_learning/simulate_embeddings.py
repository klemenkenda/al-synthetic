from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


def uncertainty_from_probs(probs: np.ndarray, strategy: str) -> np.ndarray:
    # probs: (N, C)
    if strategy == "entropy":
        ent = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
        return ent
    if strategy == "least_confidence":
        return 1.0 - probs.max(axis=1)
    if strategy == "margin":
        part = np.partition(-probs, 1, axis=1)
        top1 = -part[:, 0]
        top2 = -part[:, 1]
        return 1.0 - (top1 - top2)
    raise ValueError(f"Unknown strategy: {strategy}")


def greedy_diverse_topk(emb: np.ndarray, utility: np.ndarray, k: int) -> list[int]:
    # copied simple greedy diversity used in model-based AL
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


def load_npz(split_npz: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    arr = np.load(split_npz, allow_pickle=True)
    emb = arr["embeddings"]
    y = arr["labels"]
    paths = [str(p) for p in arr["paths"].tolist()]
    return emb, y, paths


def run_al_simulation(
    embeddings_dir: Path,
    seed_size: int = 80,
    rounds: int = 5,
    query_size: int = 50,
    strategy: str = "entropy",
    diversity: bool = False,
    out_dir: Path | None = None,
    seed: int = 42,
) -> None:
    np.random.seed(seed)
    embeddings_dir = Path(embeddings_dir)
    out_dir = Path(out_dir or (embeddings_dir.parent / "al_embeddings_results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # expected files: train_embeddings.npz, val_embeddings.npz, test_embeddings.npz
    train_npz = embeddings_dir / "train_embeddings.npz"
    test_npz = embeddings_dir / "test_embeddings.npz"
    if not train_npz.exists() or not test_npz.exists():
        raise FileNotFoundError(f"Expected embeddings in {embeddings_dir} (train/test .npz files)")

    X_train_all, y_train_all, paths_train = load_npz(train_npz)
    X_test, y_test, paths_test = load_npz(test_npz)

    n_samples = X_train_all.shape[0]
    idxs = np.arange(n_samples)
    np.random.shuffle(idxs)

    seed_size = min(seed_size, n_samples)
    labeled_idxs = list(idxs[:seed_size].tolist())
    unlabeled_idxs = list(idxs[seed_size:].tolist())

    classifiers = {
        "logreg": LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="auto"),
        "random_forest": RandomForestClassifier(n_estimators=200),
        "svc": SVC(probability=True),
        "gbdt": GradientBoostingClassifier(n_estimators=200),
    }

    for name, clf in classifiers.items():
        print(f"[RUN] classifier={name} seed_size={seed_size} rounds={rounds} query_size={query_size}")
        # copy per-run state
        labeled = labeled_idxs.copy()
        unlabeled = unlabeled_idxs.copy()
        metrics: list[dict[str, Any]] = []
        queries_all: list[dict[str, Any]] = []

        for r in range(1, rounds + 1):
            if not labeled:
                break
            X_train = X_train_all[labeled]
            y_train = y_train_all[labeled]

            clf.fit(X_train, y_train)

            # eval on test
            y_pred = clf.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred, average="macro"))

            # query step
            pool_size_before = len(unlabeled)
            if unlabeled:
                X_pool = X_train_all[unlabeled]
                # some classifiers may not implement predict_proba; we used prob-capable models
                probs = clf.predict_proba(X_pool)
                util = uncertainty_from_probs(probs, strategy=strategy)

                actual_k = min(query_size, len(unlabeled))
                if diversity:
                    chosen_local = greedy_diverse_topk(X_pool, util, actual_k)
                    chosen_idxs = [unlabeled[i] for i in chosen_local]
                else:
                    order = np.argsort(-util)
                    chosen_local = order[:actual_k].tolist()
                    chosen_idxs = [unlabeled[i] for i in chosen_local]

                # record queries
                for ui, ul_idx in enumerate(chosen_idxs):
                    queries_all.append(
                        {
                            "round": r,
                            "query_rank": ui,
                            "train_idx": int(ul_idx),
                            "filepath": paths_train[ul_idx],
                            "true_label": int(y_train_all[ul_idx]),
                            "uncertainty": float(util[chosen_local[ui]]),
                        }
                    )

                # add to labeled and remove from unlabeled
                labeled.extend(chosen_idxs)
                unlabeled = [i for i in unlabeled if i not in chosen_idxs]

            metrics.append(
                {
                    "round": r,
                    "labeled_size": len(labeled),
                    "pool_before": pool_size_before,
                    "queried": min(query_size, pool_size_before),
                    "pool_after": len(unlabeled),
                    "test_acc": acc,
                    "test_f1_macro": f1,
                }
            )

            print(f"  [round {r}] labeled={len(labeled)} pool={len(unlabeled)} acc={acc:.4f} f1={f1:.4f}")

            if not unlabeled:
                print("  pool exhausted, stopping early")
                break

        # save outputs
        metrics_csv = out_dir / f"{name}_metrics.csv"
        queries_csv = out_dir / f"{name}_queries.csv"
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()) if metrics else ["round"])
            writer.writeheader()
            writer.writerows(metrics)

        with queries_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["round", "query_rank", "train_idx", "filepath", "true_label", "uncertainty"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(queries_all)

        # save run summary
        summary = {
            "classifier": name,
            "seed_size": seed_size,
            "rounds_run": len(metrics),
            "final_labeled": metrics[-1]["labeled_size"] if metrics else seed_size,
            "final_pool": int(n_samples - (metrics[-1]["labeled_size"] if metrics else seed_size)),
            "metrics_csv": str(metrics_csv),
            "queries_csv": str(queries_csv),
        }
        with (out_dir / f"{name}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Active learning simulation using precomputed embeddings and sklearn classifiers.")
    parser.add_argument("--embeddings-dir", type=Path, default=Path("artifacts/embedding_cnn/embeddings"))
    parser.add_argument("--seed-size", type=int, default=80)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--query-size", type=int, default=50)
    parser.add_argument("--strategy", type=str, default="entropy", choices=["entropy", "margin", "least_confidence"])
    parser.add_argument("--diversity", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_al_simulation(
        embeddings_dir=args.embeddings_dir,
        seed_size=args.seed_size,
        rounds=args.rounds,
        query_size=args.query_size,
        strategy=args.strategy,
        diversity=args.diversity,
        out_dir=args.out_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
