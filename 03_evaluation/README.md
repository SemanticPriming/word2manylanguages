# Stage 3: Evaluation

Evaluates the word-by-dimension models produced by [`02_model_training/`](../02_model_training/) against replication norms, extended norms datasets, and word frequency counts, producing predictive accuracy scores per model configuration.

## 📂 Contents

- `evaluation.py`:
  - `loop_norms_vp` / `evaluate_norms_vp` / `predict_norms` — replicate the original van Paridon & Thompson norm predictions using ridge regression
  - `evaluate_norms` — predicts extended psycholinguistic norms datasets (valence, AoA, concreteness, etc.)
  - `evaluate_counts` / `predict_counts` — predicts word frequency counts

Output (`eval_results/`) feeds [`results/`](../results/) for reported results in [`05_manuscript/`](../05_manuscript/). See [`word2manylanguages_workflow.ipynb`](../word2manylanguages_workflow.ipynb) for a walk-through that drives this module alongside the other two stages.

> This module is planned for a rewrite to output pre-formatted data directly (see [`results/README.md`](../results/README.md) for why `04_postprocessing/` was removed).

## 📦 Requirements

- Python 3.10+
- `numpy`, `pandas`, `scikit-learn`, `unidecode`
