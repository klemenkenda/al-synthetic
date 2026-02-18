from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


st.set_page_config(page_title="AL Surface Defects MVP", layout="wide")

DEFAULT_DATA_ROOT = Path("data/synth_surface_defects")
DEFAULT_METADATA = DEFAULT_DATA_ROOT / "metadata.csv"
DEFAULT_LABELS = Path("artifacts/labels/human_labels.csv")
DEFAULT_TRAIN_ARTIFACTS = Path("artifacts/surface_classifier")
DEFAULT_AL_ARTIFACTS = Path("artifacts/active_learning")
DEFAULT_AL_STATE = DEFAULT_AL_ARTIFACTS / "session_state.json"
DEFAULT_AL_QUERY = DEFAULT_AL_ARTIFACTS / "current_query.csv"

CLASS_NAMES = ["none", "scratch", "pit_corrosion", "stain", "crack"]


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["filepath", "label", "annotator", "timestamp_utc"])
    return pd.read_csv(path)


def save_labels(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_labels(path)
    new_df = pd.DataFrame(rows)
    if new_df.empty:
        return
    if existing.empty:
        merged = new_df
    else:
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.sort_values("timestamp_utc").drop_duplicates(subset=["filepath"], keep="last")
    merged.to_csv(path, index=False)


def run_command_live(command: list[str], cwd: Path) -> int:
    st.write("Running command:")
    st.code(" ".join(command))
    output_box = st.empty()
    log_lines: list[str] = []

    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    for line in proc.stdout:
        log_lines.append(line.rstrip("\n"))
        # keep UI responsive; do not grow indefinitely
        output_box.code("\n".join(log_lines[-400:]), language="text")

    code = proc.wait()
    if code == 0:
        st.success("Command finished successfully.")
    else:
        st.error(f"Command failed with exit code {code}.")
    return code


def load_al_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _init_labeling_state() -> None:
    if "label_queue" not in st.session_state:
        st.session_state["label_queue"] = []
    if "pending_labels" not in st.session_state:
        st.session_state["pending_labels"] = {}
    if "batch_paths" not in st.session_state:
        st.session_state["batch_paths"] = []
    if "batch_signature" not in st.session_state:
        st.session_state["batch_signature"] = None
    if "batch_refresh_counter" not in st.session_state:
        st.session_state["batch_refresh_counter"] = 0


def _add_to_queue(filepath: str) -> None:
    queue = list(st.session_state["label_queue"])
    if filepath not in queue:
        queue.append(filepath)
    st.session_state["label_queue"] = queue


def _remove_from_queue(filepath: str) -> None:
    queue = list(st.session_state["label_queue"])
    if filepath in queue:
        queue.remove(filepath)
    st.session_state["label_queue"] = queue


def _ensure_batch(pool: pd.DataFrame, split: str, batch_size: int, seed: int) -> None:
    sig = (split, int(batch_size), int(seed), int(st.session_state["batch_refresh_counter"]))
    needs_new = (
        st.session_state["batch_signature"] != sig
        or not st.session_state["batch_paths"]
    )
    if not needs_new:
        return
    if pool.empty:
        st.session_state["batch_paths"] = []
        st.session_state["batch_signature"] = sig
        return

    sample_n = min(batch_size, len(pool))
    batch_df = pool.sample(n=sample_n, random_state=int(seed) + int(st.session_state["batch_refresh_counter"]))
    st.session_state["batch_paths"] = batch_df["filepath"].tolist()
    st.session_state["batch_signature"] = sig


def labeling_tab(
    metadata: pd.DataFrame,
    labels_path: Path,
    data_root: Path,
    focus_paths: list[str] | None = None,
    lock_to_focus: bool = False,
) -> None:
    st.subheader("Label Data")
    _init_labeling_state()
    labels_df = load_labels(labels_path)
    labeled_paths = set(labels_df["filepath"].tolist()) if not labels_df.empty else set()
    labeled_map = dict(zip(labels_df["filepath"], labels_df["label"])) if not labels_df.empty else {}

    if metadata.empty:
        st.warning("No metadata found. Generate dataset first.")
        return

    top_left, top_mid, top_right, top_extra = st.columns([2, 2, 2, 2])
    split = top_left.selectbox("Split to label", options=sorted(metadata["split"].unique().tolist()), index=0)
    batch_size = top_mid.slider("Batch size", min_value=4, max_value=32, value=12, step=4)
    seed = top_right.number_input("Batch seed", min_value=0, value=42, step=1)
    show_previously_labeled = top_extra.checkbox("Show previously labeled", value=True, disabled=lock_to_focus)

    split_all = metadata[metadata["split"] == split].copy()
    if focus_paths:
        focus_set = set(focus_paths)
        pool = split_all[split_all["filepath"].isin(focus_set)].copy()
        if lock_to_focus:
            st.info(f"Labeling is locked to current queried batch ({len(pool)} samples).")
    else:
        pool = split_all.copy()
        if not show_previously_labeled:
            pool = pool[~pool["filepath"].isin(labeled_paths)]

    st.write(
        f"In split `{split}`: total=**{len(split_all)}**, "
        f"previously labeled=**{len(split_all[split_all['filepath'].isin(labeled_paths)])}**, "
        f"currently selectable=**{len(pool)}**"
    )
    if pool.empty:
        st.info("No samples available with current filter.")
        return

    if top_right.button("Load New Batch"):
        st.session_state["batch_refresh_counter"] += 1

    _ensure_batch(pool, split, int(batch_size), int(seed))
    batch_paths: list[str] = list(st.session_state["batch_paths"])
    if not batch_paths:
        st.warning("No batch available.")
        return

    st.markdown("### Batch Labeling Workflow")
    st.caption("1) Add images to queue. 2) Choose label in sidebar. 3) Apply label to queue. 4) Save pending labels.")

    with st.sidebar:
        st.markdown("### Label Controls")
        annotator = st.text_input("Annotator", value="user")
        queue: list[str] = list(st.session_state["label_queue"])
        pending: dict[str, str] = dict(st.session_state["pending_labels"])
        active_label = st.radio("Active label", options=CLASS_NAMES, index=0, horizontal=False)
        if st.button("Apply Label to Queue", use_container_width=True):
            if not queue:
                st.warning("Queue is empty.")
            else:
                for fp in queue:
                    pending[fp] = active_label
                st.session_state["pending_labels"] = pending
                st.session_state["label_queue"] = []
                st.success(f"Applied `{active_label}` to {len(queue)} image(s).")

        st.markdown(f"**Queue size:** {len(queue)}")
        if queue:
            st.caption("Images in queue:")
            preview = pd.DataFrame({"filepath": queue[:20]})
            st.dataframe(preview, use_container_width=True, hide_index=True)
            if len(queue) > 20:
                st.caption(f"... and {len(queue) - 20} more")
        else:
            st.caption("Queue is empty.")

        if st.button("Clear Queue", use_container_width=True):
            st.session_state["label_queue"] = []
            st.info("Queue cleared.")

        pending_counts = Counter(st.session_state["pending_labels"].values())
        if pending_counts:
            st.markdown("### Pending Labels")
            for cls in CLASS_NAMES:
                st.write(f"- `{cls}`: {pending_counts.get(cls, 0)}")
        else:
            st.caption("No pending labels yet.")

    cols = st.columns(4)
    pending = dict(st.session_state["pending_labels"])
    queue_set = set(st.session_state["label_queue"])

    for i, filepath in enumerate(batch_paths):
        col = cols[i % 4]
        with col:
            image_path = data_root / filepath

            assigned = pending.get(filepath)
            previously_labeled = labeled_map.get(filepath)
            status = "IN QUEUE" if filepath in queue_set else "NOT IN QUEUE"
            chip_bg = "#374151"
            chip_txt = "Unlabeled"
            image_border = "3px solid #6b7280"
            if previously_labeled:
                chip_bg = "#14532d"
                chip_txt = f"Previously labeled: {previously_labeled}"
                image_border = "4px solid #16a34a"
            if assigned:
                chip_bg = "#78350f"
                chip_txt = f"Pending label: {assigned}"
                image_border = "4px solid #f59e0b"
            if filepath in queue_set:
                chip_bg = "#1e3a8a"
                chip_txt = f"In queue | {chip_txt}"
                image_border = "4px solid #2563eb"

            st.markdown(
                f"<div style='background:{chip_bg};color:white;padding:6px 8px;border-radius:8px;"
                f"font-size:12px;font-weight:600;margin-bottom:6px'>{chip_txt}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Queue status: {status}")

            st.image(str(image_path), caption=filepath, use_container_width=True)
            if filepath in queue_set:
                if st.button("Unselect Image", key=f"rmq_{i}_{filepath}", use_container_width=True):
                    _remove_from_queue(filepath)
                    st.rerun()
            else:
                if st.button("Select Image", key=f"queue_btn_{i}_{filepath}", use_container_width=True):
                    _add_to_queue(filepath)
                    st.rerun()

            quick = st.selectbox(
                "Quick label",
                options=[""] + CLASS_NAMES,
                index=0,
                key=f"quick_{i}_{filepath}",
            )
            if quick:
                pending[filepath] = quick
                st.session_state["pending_labels"] = pending

    st.markdown("### Save / Review")
    pending_df = pd.DataFrame(
        [{"filepath": fp, "label": lbl} for fp, lbl in st.session_state["pending_labels"].items()]
    )
    if not pending_df.empty:
        st.dataframe(pending_df, use_container_width=True)
    else:
        st.info("No pending labels to save.")

    save_col, clear_col = st.columns(2)
    if save_col.button("Save All Pending Labels", type="primary"):
        pending_now: dict[str, str] = dict(st.session_state["pending_labels"])
        if not pending_now:
            st.warning("No pending labels selected.")
        else:
            now = datetime.now(timezone.utc).isoformat()
            rows = [
                {
                    "filepath": fp,
                    "label": lbl,
                    "annotator": annotator,
                    "timestamp_utc": now,
                }
                for fp, lbl in pending_now.items()
            ]
            save_labels(labels_path, rows)
            st.session_state["pending_labels"] = {}
            st.session_state["label_queue"] = []
            st.success(f"Saved {len(rows)} labels to `{labels_path}`")
            st.rerun()
    if clear_col.button("Clear Pending Labels"):
        st.session_state["pending_labels"] = {}
        st.info("Cleared pending labels.")


def run_tab(project_root: Path, labels_path: Path) -> None:
    st.subheader("Run Experiments")

    st.markdown("### Baseline Training")
    c1, c2, c3 = st.columns(3)
    epochs = c1.number_input("Epochs", min_value=1, value=20)
    batch_size = c2.number_input("Batch size", min_value=1, value=32)
    lr = c3.number_input("Learning rate", min_value=0.00001, value=0.001, format="%.5f")
    use_labels_for_train = st.checkbox("Use human label overrides in training", value=True)

    if st.button("Run Baseline Training"):
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.train.train_baseline",
            "--data-root",
            str(DEFAULT_DATA_ROOT),
            "--epochs",
            str(int(epochs)),
            "--batch-size",
            str(int(batch_size)),
            "--lr",
            str(float(lr)),
        ]
        if use_labels_for_train and labels_path.exists():
            cmd += ["--labels-csv", str(labels_path)]
        run_command_live(cmd, cwd=project_root)

    st.markdown("---")
    st.markdown("### Active Learning Loop")
    a1, a2, a3 = st.columns(3)
    rounds = a1.number_input("Rounds", min_value=1, value=5)
    epochs_per_round = a2.number_input("Epochs per round", min_value=1, value=8)
    query_size = a3.number_input("Query size", min_value=1, value=50)

    b1, b2, b3 = st.columns(3)
    seed_size = b1.number_input("Seed size", min_value=1, value=80)
    strategy = b2.selectbox("Uncertainty strategy", options=["entropy", "margin", "least_confidence"], index=0)
    diversity = b3.checkbox("Diversity sampling", value=True)
    warm_start = st.checkbox("Warm start from baseline checkpoint", value=True)
    use_labels_for_al = st.checkbox("Use human label overrides in AL training", value=True)

    if st.button("Run Active Learning Loop"):
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.active_learning.simulate",
            "--data-root",
            str(DEFAULT_DATA_ROOT),
            "--checkpoint",
            str(DEFAULT_TRAIN_ARTIFACTS / "best.pt"),
            "--rounds",
            str(int(rounds)),
            "--epochs-per-round",
            str(int(epochs_per_round)),
            "--query-size",
            str(int(query_size)),
            "--seed-size",
            str(int(seed_size)),
            "--strategy",
            strategy,
        ]
        if diversity:
            cmd.append("--diversity")
        if warm_start:
            cmd.append("--warm-start")
        if use_labels_for_al and labels_path.exists():
            cmd += ["--labels-csv", str(labels_path)]
        run_command_live(cmd, cwd=project_root)


def workflow_tab(project_root: Path, labels_path: Path) -> None:
    st.subheader("Active Learning Workflow (Proper Loop)")
    st.caption("Round flow: initialize -> run train+query -> human label queried batch -> finalize labels -> next round")

    state = load_al_state(DEFAULT_AL_STATE)
    c1, c2, c3 = st.columns(3)
    seed_size = c1.number_input("Initial seed size", min_value=1, value=80, key="wf_seed_size")
    query_size = c2.number_input("Query size", min_value=1, value=50, key="wf_query_size")
    epochs_per_round = c3.number_input("Epochs per round", min_value=1, value=8, key="wf_epochs_round")

    d1, d2, d3 = st.columns(3)
    strategy = d1.selectbox("Strategy", options=["entropy", "margin", "least_confidence"], index=0, key="wf_strategy")
    diversity = d2.checkbox("Diversity", value=True, key="wf_diversity")
    batch_size = d3.number_input("Batch size", min_value=1, value=64, key="wf_batch_size")

    if state is None:
        st.warning("No AL session initialized.")
    else:
        st.info(
            f"Round={state.get('round', 0)} | labeled={len(state.get('labeled_paths', []))} | "
            f"unlabeled={len(state.get('unlabeled_paths', []))} | pending_query={len(state.get('pending_query_paths', []))}"
        )

    i1, i2, i3 = st.columns(3)
    if i1.button("Initialize AL Session", type="primary"):
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.active_learning.session_loop",
            "--action",
            "init",
            "--data-root",
            str(DEFAULT_DATA_ROOT),
            "--state-json",
            str(DEFAULT_AL_STATE),
            "--seed-size",
            str(int(seed_size)),
            "--seed",
            "42",
            "--query-size",
            str(int(query_size)),
            "--epochs-per-round",
            str(int(epochs_per_round)),
            "--strategy",
            strategy,
        ]
        if diversity:
            cmd.append("--diversity")
        run_command_live(cmd, cwd=project_root)
        st.rerun()

    if i2.button("Run Next Round (Train + Query)"):
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.active_learning.session_loop",
            "--action",
            "run_round",
            "--data-root",
            str(DEFAULT_DATA_ROOT),
            "--state-json",
            str(DEFAULT_AL_STATE),
            "--query-csv",
            str(DEFAULT_AL_QUERY),
            "--metrics-csv",
            str(DEFAULT_AL_ARTIFACTS / "al_metrics.csv"),
            "--labels-csv",
            str(labels_path),
            "--query-size",
            str(int(query_size)),
            "--epochs-per-round",
            str(int(epochs_per_round)),
            "--batch-size",
            str(int(batch_size)),
            "--strategy",
            strategy,
        ]
        if diversity:
            cmd.append("--diversity")
        run_command_live(cmd, cwd=project_root)
        st.rerun()

    if i3.button("Finalize Queried Labels"):
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.active_learning.session_loop",
            "--action",
            "finalize_query",
            "--data-root",
            str(DEFAULT_DATA_ROOT),
            "--state-json",
            str(DEFAULT_AL_STATE),
            "--labels-csv",
            str(labels_path),
        ]
        run_command_live(cmd, cwd=project_root)
        st.rerun()

    if DEFAULT_AL_QUERY.exists():
        st.markdown("### Current Queried Batch")
        try:
            st.dataframe(pd.read_csv(DEFAULT_AL_QUERY), use_container_width=True)
        except Exception:
            st.warning("Could not read current query CSV.")


def metrics_tab() -> None:
    st.subheader("Metrics Dashboard")
    st.caption("Refresh this tab after each run to load latest metrics.")

    training_metrics = DEFAULT_TRAIN_ARTIFACTS / "training_metrics.csv"
    al_metrics = DEFAULT_AL_ARTIFACTS / "al_metrics.csv"
    al_epoch_history = DEFAULT_AL_ARTIFACTS / "al_epoch_history.csv"

    st.markdown("### Baseline Training Metrics")
    if training_metrics.exists():
        tm = pd.read_csv(training_metrics)
        st.line_chart(tm.set_index("epoch")[["train_acc", "val_acc"]])
        st.line_chart(tm.set_index("epoch")[["train_loss", "val_loss"]])
    else:
        st.info("No `training_metrics.csv` found yet.")

    st.markdown("### Active Learning Metrics")
    if al_metrics.exists():
        am = pd.read_csv(al_metrics)
        st.line_chart(am.set_index("round")[["best_val_acc", "test_acc"]])
        st.dataframe(am, use_container_width=True)
    else:
        st.info("No `al_metrics.csv` found yet.")

    st.markdown("### AL Epoch History")
    if al_epoch_history.exists():
        eh = pd.read_csv(al_epoch_history)
        st.dataframe(eh, use_container_width=True)
    else:
        st.info("No `al_epoch_history.csv` found yet.")


def main() -> None:
    st.title("Active Learning MVP: Surface Defects")

    project_root = Path.cwd()
    data_root = DEFAULT_DATA_ROOT
    metadata_path = DEFAULT_METADATA
    labels_path = DEFAULT_LABELS

    metadata = load_metadata(metadata_path)
    state = load_al_state(DEFAULT_AL_STATE)
    pending_query = state.get("pending_query_paths", []) if state else []

    tab1, tab2, tab3, tab4 = st.tabs(["Label Data", "AL Workflow", "Run Experiments", "Metrics"])
    with tab1:
        labeling_tab(
            metadata,
            labels_path,
            data_root,
            focus_paths=pending_query if pending_query else None,
            lock_to_focus=bool(pending_query),
        )
    with tab2:
        workflow_tab(project_root, labels_path)
    with tab3:
        run_tab(project_root, labels_path)
    with tab4:
        metrics_tab()


if __name__ == "__main__":
    main()
