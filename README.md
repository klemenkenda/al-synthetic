# Synthetic Textured Surface Defect Dataset

This project generates a synthetic industrial inspection dataset and runs classification + active learning experiments.

## Dataset Spec
- Image size: `256x256` grayscale
- Classes: `none`, `scratch`, `pit_corrosion`, `stain`, `crack`
- Defect parameters: `size`, `contrast`, `blur`, `density`, `orientation`
- Segmentation masks: optional (enabled by default in config)

## 1) Environment Setup
Create and activate a dedicated conda environment:

```powershell
conda create -n al-synth python=3.11 -y
conda activate al-synth
# Use the provided requirements file to install Python deps.
python -m pip install -r requirements.txt
# If you require a CUDA-enabled PyTorch, follow the official selector and install
# PyTorch with the appropriate CUDA runtime (https://pytorch.org/get-started/locally/).
```

## 2) Generate Dataset
Generate 1,000 images (old output is removed automatically before generation):

```powershell
python -m src.pipeline.generate_dataset --config config/dataset_config.json --output data/synth_surface_defects --num-images 1000 --seed 42
```

Alternative:
```powershell
./scripts/generate_dataset.ps1
```

Expected dataset outputs:
- Images: `data/synth_surface_defects/{train|val|test}/{class}/img_XXXXX.png`
- Masks: `data/synth_surface_defects/masks/{train|val|test}/{class}/img_XXXXX.png`
- Metadata: `data/synth_surface_defects/metadata.csv`
- Summary: `data/synth_surface_defects/summary.json`

## 3) Run Standard Classification Training
Train baseline classifier:

```powershell
python -m src.train.train_baseline --data-root data/synth_surface_defects --epochs 20 --batch-size 32 --lr 0.001
```

Training outputs (normal training metrics):
- Model checkpoint: `artifacts/surface_classifier/best.pt`
- Epoch metrics: `artifacts/surface_classifier/training_metrics.csv`
- Run summary: `artifacts/surface_classifier/training_run_summary.json`
- Test uncertainty/export: `artifacts/surface_classifier/test_uncertainty_scores.csv`

### Train Separate Embedding CNN (for feature extraction)
This trains a CNN classifier and exports bottleneck embeddings (recommended for similarity, clustering, AL diversity):

```powershell
python -m src.train.train_embedding_cnn --data-root data/synth_surface_defects --epochs 20 --batch-size 32 --emb-dim 256
```

Outputs:
- checkpoint: `artifacts/embedding_cnn/best.pt`
- training metrics: `artifacts/embedding_cnn/training_metrics.csv`
- summary: `artifacts/embedding_cnn/run_summary.json`
- embeddings:
  - `artifacts/embedding_cnn/embeddings/train_embeddings.npz`
  - `artifacts/embedding_cnn/embeddings/val_embeddings.npz`
  - `artifacts/embedding_cnn/embeddings/test_embeddings.npz`
  - index files `*_embeddings_index.csv`

## Embeddings-based Active Learning Simulations

Use precomputed embeddings with lightweight sklearn classifiers to simulate AL rounds (logistic regression, random forest, SVC, gradient boosting).

Example usage:

```powershell
python -m src.active_learning.simulate_embeddings --embeddings-dir artifacts/embedding_cnn/embeddings --seed-size 80 --rounds 5 --query-size 50
```

Options:
- `--embeddings-dir`: directory containing `*_embeddings.npz` files (default `artifacts/embedding_cnn/embeddings`).
- `--seed-size`, `--rounds`, `--query-size`, `--strategy` (`entropy`|`margin`|`least_confidence`), `--diversity`, `--out-dir`, `--seed`.

Dependencies:
- See `requirements.txt` for the baseline Python packages used by the project.

Install into your conda env using the repository requirements:

```powershell
conda activate al-synth
python -m pip install -r requirements.txt
```

Outputs (default `artifacts/embedding_cnn/al_embeddings_results/`):
- `{classifier}_metrics.csv` — per-round test accuracy/F1 and pool sizes
- `{classifier}_queries.csv` — query history across rounds
- `{classifier}_summary.json` — run summary and paths

Note: ensure embeddings exist before running (see the embedding CNN step above).

## 7) Comprehensive AL Experiments

Run a full suite of AL experiments comparing:
- **Classifiers**: Logistic Regression, Random Forest, SVC, Gradient Boosting
- **Sampling Strategies**: Entropy, Margin, Least Confidence
- **Diversity**: With and without embedding-space diversity
- **Baselines**: Train on full dataset for comparison

### Run All Experiments

```powershell
python scripts/run_al_experiments.py --embeddings-dir artifacts/embedding_cnn/embeddings --seed-size 1 --rounds 50 --query-size 1
```

Options:
- `--seed-size`: initial labeled set size (default: 1)
- `--rounds`: number of AL rounds (default: 50)
- `--query-size`: samples per query (default: 1)
- `--out-dir`: output directory for results (default: `artifacts/embedding_cnn/al_experiments`)

### Outputs
- `baseline_results.csv` — performance of each classifier trained on full training set
- `all_al_curves.csv` — per-round test accuracy and F1 for all strategy-classifier combinations
- `experiment_metadata.json` — experiment configuration and timestamp
- Per-strategy subdirectories with individual classifier metrics

### Analyze Results with Jupyter Notebook

After running experiments, open and run the analysis notebook:

```powershell
jupyter notebook notebooks/al_analysis.ipynb
```

The notebook includes:
1. Load baseline and AL experimental data
2. Baseline metrics visualization
3. Learning curves for each classifier across all strategies
4. Comparison of sampling strategies
5. Heatmap: final accuracy by strategy and classifier
6. Summary statistics and rankings
7. Export results and generate analysis report

All plots are saved to `artifacts/embedding_cnn/al_experiments/` for reporting.

## 5) Run Full-Loop Active Learning
Run iterative AL rounds (train -> query -> add queried samples -> retrain):

```powershell
python -m src.active_learning.simulate --data-root data/synth_surface_defects --checkpoint artifacts/surface_classifier/best.pt --rounds 5 --epochs-per-round 8 --strategy entropy --query-size 50 --seed-size 80 --diversity
```

Active learning outputs (separate AL metrics):
- Query history (all rounds): `artifacts/active_learning/al_query.csv`
- Round metrics: `artifacts/active_learning/al_metrics.csv`
- Epoch history (per round): `artifacts/active_learning/al_epoch_history.csv`
- AL run summary: `artifacts/active_learning/al_run_summary.json`

## 6) Useful Variants
Run with margin uncertainty:

```powershell
python -m src.active_learning.simulate --data-root data/synth_surface_defects --checkpoint artifacts/surface_classifier/best.pt --rounds 5 --epochs-per-round 8 --strategy margin --query-size 50 --seed-size 80
```

Run with least-confidence uncertainty:

```powershell
python -m src.active_learning.simulate --data-root data/synth_surface_defects --checkpoint artifacts/surface_classifier/best.pt --rounds 5 --epochs-per-round 8 --strategy least_confidence --query-size 50 --seed-size 80
```

Use warm-start from the pretrained baseline weights in round 1:

```powershell
python -m src.active_learning.simulate --data-root data/synth_surface_defects --checkpoint artifacts/surface_classifier/best.pt --warm-start --rounds 5 --epochs-per-round 8 --strategy entropy --query-size 50 --seed-size 80
```

## 7) Streamlit MVP App
Launch the app:

```powershell
streamlit run app.py
```

What it provides:
- Batch labeling UI (multiple images at once)
- Proper AL workflow tab:
  - initialize AL session
  - run next round (train + query)
  - label queried batch
  - finalize labels to move them into labeled pool
- Metrics dashboard:
  - baseline: train/val accuracy and loss by epoch
  - AL: test/best-val accuracy by round

App label storage:
- Human labels are saved to: `artifacts/labels/human_labels.csv`
- Training/AL can use these labels as overrides.

## Notes
- If `torch` is not installed, training and AL scripts will fail to import.
- Dataset generation and training are deterministic for a fixed `--seed`.
