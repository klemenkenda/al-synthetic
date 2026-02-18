from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.generator.scene import SceneSpec, apply_defect, create_base_surface, sample_params, serializable_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate textured-surface defect dataset.")
    parser.add_argument("--config", type=Path, default=Path("config/dataset_config.json"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_split(rng: random.Random, splits: dict[str, float]) -> str:
    r = rng.random()
    t = splits["train"]
    v = t + splits["val"]
    if r < t:
        return "train"
    if r < v:
        return "val"
    return "test"


def choose_class(rng: random.Random, class_weights: dict[str, float]) -> str:
    labels = list(class_weights.keys())
    weights = [float(class_weights[k]) for k in labels]
    total = sum(max(0.0, w) for w in weights)
    if total <= 0:
        return labels[0]

    r = rng.uniform(0.0, total)
    acc = 0.0
    for label, weight in zip(labels, weights):
        acc += max(0.0, weight)
        if r <= acc:
            return label
    return labels[-1]


def clear_output_dir(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)


def ensure_dirs(root: Path, classes: list[str], export_masks: bool) -> None:
    for split in ["train", "val", "test"]:
        for cls in classes:
            (root / split / cls).mkdir(parents=True, exist_ok=True)
            if export_masks:
                (root / "masks" / split / cls).mkdir(parents=True, exist_ok=True)


def _print_progress(generated: int, total: int, started_at: float, class_counts: Counter[str], split_counts: Counter[str]) -> None:
    elapsed = max(time.time() - started_at, 1e-9)
    rate = generated / elapsed
    eta_sec = int(max(total - generated, 0) / max(rate, 1e-9))
    by_class = " ".join(f"{k}={class_counts[k]}" for k in sorted(class_counts.keys()))
    print(
        (
            f"[progress] {generated}/{total} "
            f"({(generated / max(total, 1)) * 100:.1f}%) "
            f"| {rate:.2f} img/s | ETA {eta_sec}s "
            f"| train={split_counts['train']} val={split_counts['val']} test={split_counts['test']} "
            f"| {by_class}"
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    width = int(cfg["image"]["width"])
    height = int(cfg["image"]["height"])
    num_images = int(args.num_images if args.num_images is not None else cfg["dataset"]["num_images"])
    classes = list(cfg["classes"])
    class_weights = dict(cfg["class_weights"])
    export_masks = bool(cfg["output"].get("export_masks", True))

    out_root = args.output if args.output is not None else Path(cfg["output"]["root"])
    out_root = Path(out_root)
    clear_output_dir(out_root)
    ensure_dirs(out_root, classes, export_masks)

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    started_at = time.time()
    progress_every = max(1, min(100, num_images // 10))

    print(f"Starting generation: target={num_images}, output='{out_root}'", flush=True)

    for i in range(num_images):
        defect_type = choose_class(rng, class_weights)
        params = sample_params(rng) if defect_type != "none" else {"size": 0, "contrast": 0, "blur": 0, "density": 0, "orientation": 0}

        base = create_base_surface(SceneSpec(width=width, height=height), rng)
        image, mask = apply_defect(base, defect_type, params, rng)
        split = choose_split(rng, cfg["dataset"]["splits"])

        filename = f"img_{i:05d}.png"
        rel_img_path = Path(split) / defect_type / filename
        full_img_path = out_root / rel_img_path
        image.save(full_img_path)

        rel_mask_path = ""
        if export_masks:
            rel_mask = Path("masks") / split / defect_type / filename
            rel_mask_path = str(rel_mask).replace("\\", "/")
            mask.save(out_root / rel_mask)

        class_counts[defect_type] += 1
        split_counts[split] += 1

        rows.append(
            {
                "index": i,
                "split": split,
                "filepath": str(rel_img_path).replace("\\", "/"),
                "mask_path": rel_mask_path,
                "label": defect_type,
                "params": json.dumps(serializable_params(params)),
                "seed": rng.randint(0, 10_000_000),
            }
        )

        generated = i + 1
        if generated % progress_every == 0 or generated == num_images:
            _print_progress(generated, num_images, started_at, class_counts, split_counts)

    with (out_root / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_images": num_images,
        "class_breakdown": {cls: class_counts[cls] for cls in classes},
        "split_breakdown": {s: split_counts[s] for s in ["train", "val", "test"]},
    }

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
